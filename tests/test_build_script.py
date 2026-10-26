"""Build-wrapper regressions; run with python3 -m unittest discover -s tests -p test_build_script.py."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'build.sh'


class BuildScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='build-script-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source with spaces'
        self.source.mkdir()
        shutil.copy2(SCRIPT, self.source / 'build.sh')
        (self.source / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.24)\nproject(probe NONE)\n'
            'option(DMFE_PORTABLE_BUILD "portable" OFF)\n')
        self.env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL='2')

    def run_build(self, *args, source=None, success=True):
        result = subprocess.run(['bash', str((source or self.source) / 'build.sh'), *args],
                                env=self.env, cwd=self.root, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def test_copied_source_and_moved_build_reconfigure(self):
        self.run_build('--cuda=off')
        copied = self.root / 'copy'
        shutil.copytree(self.source, copied)
        self.assertIn('clearing stale', self.run_build('--cuda=off', source=copied))
        moved = self.source / 'build' / 'moved'
        shutil.copytree(self.source / 'build', moved)
        self.assertIn('clearing stale', self.run_build('moved', '--cuda=off'))
        self.assertNotIn('clearing stale', self.run_build('moved', '--cuda=off'))

    def test_alias_keeps_custom_options(self):
        self.run_build('--cuda=off')
        cache = self.source / 'build' / 'CMakeCache.txt'
        cache.write_text(cache.read_text().replace('DMFE_PORTABLE_BUILD:BOOL=OFF',
                                                 'DMFE_PORTABLE_BUILD:BOOL=ON'))
        alias = self.root / 'alias'
        alias.symlink_to(self.source, target_is_directory=True)
        self.assertNotIn('clearing stale', self.run_build('--cuda=off', source=alias))
        build_alias = self.root / 'build-alias'
        build_alias.symlink_to(self.source / 'build', target_is_directory=True)
        self.run_build(str(build_alias), '--cuda=off')
        self.assertIn('DMFE_PORTABLE_BUILD:BOOL=ON', cache.read_text())

    def test_cache_created_through_alias_is_kept(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.source, target_is_directory=True)
        result = subprocess.run(['cmake', '-S', str(alias), '-B', str(alias / 'build'),
                                 '-DDMFE_PORTABLE_BUILD=ON'], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn('clearing stale', self.run_build('--cuda=off'))
        self.assertIn('DMFE_PORTABLE_BUILD:BOOL=ON',
                      (self.source / 'build' / 'CMakeCache.txt').read_text())

    def test_clean_cannot_remove_sources(self):
        self.run_build(str(self.source), '--clean', '--cuda=off', success=False)
        self.run_build(str(self.root), '--clean', '--cuda=off', success=False)
        self.assertTrue((self.source / 'CMakeLists.txt').is_file())

    def mocks(self):
        # Isolate discovery from the workstation's compilers, GPUs and modules.
        bindir = self.root / 'bin'
        bindir.mkdir()
        for name in ('bash', 'dirname', 'mkdir', 'sed', 'rm', 'date', 'mv', 'head', 'tr'):
            (bindir / name).symlink_to(shutil.which(name))
        self.log = self.root / 'arguments'
        self.env.update(PATH=str(bindir), ARG_LOG=str(self.log))
        self.executable(bindir / 'cmake', '#!/bin/sh\nprintf "%s\\n" "$@" >> "$ARG_LOG"\n')
        self.executable(bindir / 'sysctl', '#!/bin/sh\necho 3\n')
        return bindir

    @staticmethod
    def executable(path, content):
        path.write_text(content)
        path.chmod(0o755)

    def test_exported_architectures_and_macos_parallelism(self):
        bindir = self.mocks()
        self.executable(bindir / 'nvcc', '#!/bin/sh\necho "release 12.5"\n')
        self.env['CMAKE_CUDA_ARCHITECTURES'] = '70;80'
        self.env.pop('CMAKE_BUILD_PARALLEL_LEVEL')
        self.run_build('--cuda=on')
        arguments = self.log.read_text().splitlines()
        self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=70;80', arguments)
        self.assertEqual(arguments[-2:], ['-j', '3'])

    def test_modules_before_cuda_detection(self):
        bindir = self.mocks()
        cuda_bin = self.root / 'cuda'
        cuda_bin.mkdir()
        self.executable(cuda_bin / 'nvcc', '#!/bin/sh\necho "release 12.5"\n')
        # BASH_ENV simulates the cluster's module shell function.
        init = self.root / 'modules.sh'
        init.write_text('module() { export PATH="$CUDA_TEST_BIN:$PATH"; }\n')
        self.env.update(BASH_ENV=str(init), CUDA_TEST_BIN=str(cuda_bin))
        for mode in ('on', 'auto'):
            with self.subTest(mode=mode):
                self.run_build('--cuda=' + mode)
                self.assertIn('-DDMFE_WITH_CUDA=ON', self.log.read_text())
                self.log.unlink()
        self.run_build('--cuda=off')
        self.assertIn('-DDMFE_WITH_CUDA=OFF', self.log.read_text())
        self.assertNotIn('[env] Trying', self.run_build('--cuda=off'))

    def test_arguments_are_validated_and_separator_is_honored(self):
        self.mocks()
        self.assertIn('Unknown option', self.run_build('--cuda=of', success=False))
        self.assertIn('Only one build directory',
                      self.run_build('one', 'two', success=False))
        self.run_build('--cuda=off', '--', '--clean')
        self.assertIn(str(self.source.resolve() / 'build' / '--clean'),
                      self.log.read_text().splitlines())

    def test_explicit_compilers_are_not_overridden(self):
        bindir = self.mocks()
        for name in ('clang-14', 'clang++-14'):
            self.executable(bindir / name, '#!/bin/sh\nexit 0\n')
        self.env.update(CC='/site/bin/cc', CXX='/site/bin/c++',
                        CUDAHOSTCXX='/site/bin/c++')
        self.run_build('--cuda=off')
        self.assertNotIn('-DCMAKE_CXX_COMPILER=clang++-14', self.log.read_text())
        self.executable(bindir / 'nvcc', '#!/bin/sh\necho "release 12.5"\n')
        self.run_build('--cuda=on')
        self.assertIn('-DCMAKE_CUDA_HOST_COMPILER=/site/bin/c++', self.log.read_text())

    def test_cached_compilers_and_architectures_are_not_overridden(self):
        bindir = self.mocks()
        for name in ('clang-14', 'clang++-14', 'nvc', 'nvc++'):
            self.executable(bindir / name, '#!/bin/sh\nexit 0\n')
        self.executable(bindir / 'nvcc', '#!/bin/sh\necho "release 12.5"\n')
        build = self.source / 'build'
        build.mkdir()
        (build / 'CMakeCache.txt').write_text(
            'CMAKE_HOME_DIRECTORY:INTERNAL=' + str(self.source.resolve()) + '\n'
            'CMAKE_CACHEFILE_DIR:INTERNAL=' + str(build.resolve()) + '\n'
            'CMAKE_CUDA_ARCHITECTURES:STRING=70\n'
            'CMAKE_CXX_COMPILER:FILEPATH=/site/bin/c++\n'
            'CMAKE_CUDA_COMPILER:FILEPATH=/site/bin/nvcc\n')
        self.run_build('--cuda=on')
        args = self.log.read_text().splitlines()
        self.assertFalse(any(a.startswith('-DCMAKE_CXX_COMPILER=') for a in args))
        self.assertFalse(any(a.startswith('-DCMAKE_CUDA_COMPILER=') for a in args))
        self.assertFalse(any(a.startswith('-DCMAKE_CUDA_ARCHITECTURES=') for a in args))
        self.env['CMAKE_CUDA_ARCHITECTURES'] = '80;90'
        self.run_build('--cuda=on')
        self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=80;90', self.log.read_text())

    def test_cuda_targets_match_gpu_and_toolkit(self):
        bindir = self.mocks()
        self.executable(bindir / 'nvcc',
                        '#!/bin/sh\nif [ "$1" = --list-gpu-code ]; then\n'
                        'printf "%s\\n" "$MOCK_SUPPORTED"\n'
                        'else echo "release 12.5"; fi\n')
        self.executable(bindir / 'nvidia-smi',
                        '#!/bin/sh\ncase "$1" in\n'
                        '--query-gpu=compute_cap) printf "%s\\n" "$MOCK_CAP";;\n'
                        '*) printf "%s\\n" "$MOCK_NAME";;\nesac\n')
        cases = [
            ('8.6', 'NVIDIA A10', 'sm_80\nsm_86', '86'),
            ('', 'NVIDIA A10', 'sm_80\nsm_86', '86'),
            ('', 'NVIDIA A100', 'sm_80\nsm_86', '80'),
            ('', '', 'sm_80\nsm_86', '80;86'),
            ('', '', 'sm_75', '75'),
        ]
        for capability, name, supported, expected in cases:
            with self.subTest(capability=capability, name=name, supported=supported):
                self.env.update(MOCK_CAP=capability, MOCK_NAME=name,
                                MOCK_SUPPORTED=supported)
                self.run_build('--cuda=on')
                self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=' + expected,
                              self.log.read_text().splitlines())
                self.log.unlink()
        self.env.update(MOCK_CAP='9.0', MOCK_SUPPORTED='sm_80\nsm_86')
        self.assertIn('does not support detected GPU',
                      self.run_build('--cuda=on', success=False))

    def test_required_cuda_fails_without_toolchain(self):
        self.mocks()
        self.assertIn('--cuda=on specified', self.run_build('--cuda=on', success=False))


if __name__ == '__main__':
    unittest.main()
