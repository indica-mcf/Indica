{
  description = "Integrated Diagnostic Analysis framework";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, poetry2nix }:
    let
      supportedSystems = [ "x86_64-linux" "x86_64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});
    in {
      packages = forAllSystems (system:
        with pkgs.${system}; rec {
          indica = callPackage ./default.nix {
            python = python39;
            poetry2nix = poetry2nix.lib.mkPoetry2Nix { pkgs = pkgs.${system}; };
          };
          python = (python39.withPackages (ps: [ ps.tkinter indica ]));
          default = indica;
        });

      apps = forAllSystems (system:
        with self.packages.${system}; {
          default = {
            type = "app";
            program = "${python}/bin/ipython3";
          };
        });

      devShells = forAllSystems (system:
        with pkgs.${system}; {
          python = mkShellNoCC {
            packages = [ self.packages.${system}.python poetry pre-commit ];
          };
          default = self.devShells.${system}.python;
        });
    };
}
