(* ::Package:: *)
(* DYNAMITE HDF5 Importer for Mathematica/Wolfram Language *)

BeginPackage["DYNAMITE`"]

ImportDMFE::usage = "ImportDMFE[file] imports a DYNAMITE data.h5 file and returns an Association with datasets, attributes, and convenience reshaped arrays."

Begin["`Private`"]

(* Normalize attribute values: unwrap singletons, convert ByteArray to String *)
Clear[normalizeAttrValue]
normalizeAttrValue[val_] := Module[{v = val},
  If[ListQ[v] && Length[v] == 1, v = First[v]];
  Which[
    Head[v] === ByteArray, FromCharacterCode[Normal[v]],
    True, v
  ]
]

(* Read all file-level attributes once, robust to shapes returned by Import *)
Clear[getAllAttributes]
getAllAttributes[file_] := Module[{raw, root, assoc},
  raw = Quiet@Check[Import[file, {"HDF5", "Attributes"}], <||>];
  root = Which[
    AssociationQ[raw] && KeyExistsQ[raw, "/"], raw["/"],
    AssociationQ[raw], raw,
    MatchQ[raw, {(_Rule | _RuleDelayed) ..}], Association@raw,
    MatchQ[raw, {{_, _} ..}], Association[Rule @@@ raw],
    True, <||>
  ];
  assoc = Association@KeyValueMap[#1 -> normalizeAttrValue[#2] &, root];
  assoc
]

(* Coerce numerics that should be integers *)
Clear[toInt]
toInt[v_] := Which[
  IntegerQ[v], v,
  NumericQ[v] && Round[v] == v, Round[v],
  True, v
]

(* Get list of dataset names present in the file *)
Clear[getDatasetNames]
getDatasetNames[file_] := Module[{names},
  names = Quiet@Check[Import[file, {"HDF5", "Datasets"}], {}];
  If[AssociationQ[names], Keys@names, names]
]

(* Sanitize dataset logical names: drop leading slash and group prefixes *)
Clear[sanitizeDSName]
sanitizeDSName[name_String] := Module[{s = StringReplace[name, StartOfString ~~ "/" -> ""]}, Last@StringSplit[s, "/"]]
sanitizeDSName[x_] := x

Clear[reshapeIfPossible]
reshapeIfPossible[data_List, nT_Integer?Positive, len_Integer?Positive] :=
  If[Length[data] == nT*len, Partition[data, len], data]
reshapeIfPossible[data_, ___] := data

Clear[ImportDMFE]
Options[ImportDMFE] = {"Reshape" -> True};
ImportDMFE[file_String?FileExistsQ, OptionsPattern[]] := Module[
  {attrs, lenAttr, len, nT, datasets, getDS, reshape = TrueQ@OptionValue["Reshape"], assoc, present, qkvLen, t1gridLen, rvecLen},

  (* Attributes *)
  attrs = getAllAttributes[file];

  (* Prefer attribute len if available; otherwise deduce from datasets *)
  lenAttr = Lookup[attrs, "len", Missing["NotAvailable"]];
  lenAttr = If[ListQ[lenAttr] && Length[lenAttr] == 1, First[lenAttr], lenAttr];
  lenAttr = toInt[lenAttr];

  (* Build map from sanitized logical name -> actual path in file *)
  Module[{allPaths = getDatasetNames[file], pathToKey, keyToPath},
    pathToKey = Association@Map[# -> sanitizeDSName[#] &, allPaths];
    (* Prefer the shortest path if duplicates map to same key *)
    keyToPath = Association@Normal@GroupBy[Normal@pathToKey, Last -> First, First];
    present = AssociationThread[Keys[keyToPath] -> True];
    (* Override getDS to use actual stored path *)
    getDS[name_] := Module[{p = Lookup[keyToPath, name, Missing["NA"]]},
      If[p === Missing["NA"], Missing["NotAvailable"], Quiet@Check[Import[file, {"HDF5", "Datasets", p}], Missing["NotAvailable"]]]
    ];
    (* Size helpers rely on paths too *)
    qkvLen = If[TrueQ@Lookup[present, "QKv", False],
      With[{dims = Quiet@Check[Import[file, {"HDF5", "Datasets", Lookup[keyToPath, "QKv"], "Dimensions"}], {}]}, If[ListQ[dims] && dims =!= {}, First@dims, Missing["NA"]]],
      Missing["NA"]
    ];
    t1gridLen = If[TrueQ@Lookup[present, "t1grid", False],
      With[{v = Quiet@Check[Import[file, {"HDF5", "Datasets", Lookup[keyToPath, "t1grid"]}], {}]}, If[ListQ[v], Length[v], Missing["NA"]]],
      Missing["NA"]
    ];
    rvecLen = If[TrueQ@Lookup[present, "rvec", False],
      With[{v = Quiet@Check[Import[file, {"HDF5", "Datasets", Lookup[keyToPath, "rvec"]}], {}]}, If[ListQ[v], Length[v], Missing["NA"]]],
      Missing["NA"]
    ];
  ];

  (* nT and len helpers already computed above using discovered paths *)

  len = Which[
    IntegerQ[lenAttr] && lenAttr > 0, lenAttr,
    IntegerQ[rvecLen] && rvecLen > 0, rvecLen,
    True, Missing["NotAvailable"]
  ];
  nT = Which[
    IntegerQ[t1gridLen] && t1gridLen > 0, t1gridLen,
    IntegerQ[len] && IntegerQ[qkvLen] && len > 0 && qkvLen >= len && Mod[qkvLen, len] == 0, qkvLen/len,
    True, Missing["NotAvailable"]
  ];

  (* getDS is defined above using discovered actual path *)

  datasets = Association@Map[
    # -> getDS[#] &,
    {"QKv","QRv","dQKv","dQRv","t1grid","rvec","drvec"}
  ];

  If[reshape && IntegerQ[len] && len > 0 && IntegerQ[nT] && nT > 0,
    datasets = Association@KeyValueMap[
      (#1 -> If[MemberQ[{"QKv","QRv","dQKv","dQRv"}, #1], reshapeIfPossible[#2, nT, len], #2]) &,
      datasets
    ];
  ];

  assoc = <|
    "Attributes" -> attrs,
    "Dimensions" -> <|"len" -> len, "nTimes" -> nT|>,
    "Data" -> datasets
  |>;

  assoc
]
ImportDMFE[___] := (Message[ImportDMFE::argx, "Provide a valid HDF5 file path."]; $Failed)

End[]
EndPackage[]
