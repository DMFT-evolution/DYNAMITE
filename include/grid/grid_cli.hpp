#pragma once

#include <string>

namespace dmfe {
// Handle the `grid` subcommand. Returns true if handled and sets exitCode accordingly.
bool maybe_handle_grid_cli(int argc, char** argv, int& exitCode);
}
