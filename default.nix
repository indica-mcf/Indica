{ lib, pkgs, poetry2nix, python ? pkgs.python39, ... }:
with python.pkgs;
with poetry2nix;
let
  poetryEnv = mkPoetryPackages {
    inherit python;
    projectDir = ./.;
    preferWheels = true;
    overrides = defaultPoetryOverrides.extend (self: super: {
      corner = super.corner.overridePythonAttrs (old: {
        buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
      });
    });
  };
in buildPythonPackage {
  pname = "indica";
  version = "0.1.0";
  format = "pyproject";
  src = ./.;
  nativeBuildInputs = [ poetry-core ];
  propagatedBuildInputs = poetryEnv.poetryPackages;
  doCheck = false;
  meta = with lib; {
    description = "Integrated Diagnostic Composition Analysis";
    homepage = "https://github.com/indica-mcf/indica";
    license = with licenses; [ gpl3 ];
  };
}
