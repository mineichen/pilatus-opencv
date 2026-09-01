{
  description = "Deterministic Rust + WASM + Tailwind dev shell";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      perSystem = { system, pkgs, ... }:
        let
          rust = with inputs.fenix.packages.${system}; combine [
            stable.toolchain
            targets.wasm32-unknown-unknown.stable.rust-std
            targets.aarch64-unknown-linux-gnu.stable.rust-std
          ];
          envVars = {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.llvmPackages.libclang.lib
              pkgs.opencv
              pkgs.stdenv.cc.cc.lib
            ];
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            PKG_CONFIG_PATH = pkgs.lib.makeSearchPath "lib/pkgconfig" [
              pkgs.glib.dev
              pkgs.aravis.dev
            ];
            CPATH = "${pkgs.glibc.dev}/include";
            CPLUS_INCLUDE_PATH = pkgs.lib.concatStringsSep ":" [
              "${pkgs.stdenv.cc.cc}/include/c++/${pkgs.lib.getVersion pkgs.stdenv.cc.cc}"
              "${pkgs.stdenv.cc.cc}/include/c++/${pkgs.lib.getVersion pkgs.stdenv.cc.cc}/${pkgs.stdenv.hostPlatform.config}"
            ];
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
          };
          commonBuildInputs = [
            rust
            pkgs.stdenv.cc
            pkgs.llvmPackages.libclang
            pkgs.cmake
            pkgs.just
            pkgs.opencv
            pkgs.pkg-config
            pkgs.git
          ];
          greet = ''
            echo "===================================="
            echo " Welcome to the deterministic dev shell! "
            echo "===================================="
            rustc --version && cargo --version && trunk --version
          '';
          policy = pkgs.writeText "policy.json" ''{"default":[{"type":"insecureAcceptAnything"}]}'';
          containername = "pilatus-opencv-isolated-dev";
          podmanRun = "${pkgs.podman}/bin/podman run --rm -it "
            + "--network=slirp4netns "
            + "--tmpfs /tmp "
            + "-v pilatus-opencv:/workspace/pilatus-opencv:z "
            + "-e HOME=/root "
            + "${containername}:latest /bin/entrypoint.sh";
        in
        {
          devShells.default = pkgs.mkShell({
            buildInputs = commonBuildInputs;
            shellHook = greet;
          } // envVars);
          packages.isolated-build = pkgs.dockerTools.buildImage {
            name = containername;
            tag = "latest";
            copyToRoot = pkgs.buildEnv {
              name = containername;
              paths = commonBuildInputs ++ [
                pkgs.bashInteractive
                pkgs.ripgrep
                pkgs.git
                pkgs.opencode
                pkgs.busybox
                (pkgs.writeScriptBin "entrypoint.sh" ''
                  #!${pkgs.bashInteractive}/bin/bash
                  ${greet}
                  exec ${pkgs.bashInteractive}/bin/bash
                '')
              ];
              pathsToLink = [ "/bin" "/lib" "/include" "/share" ];
            };
            config = {
              Env = pkgs.lib.mapAttrsToList (k: v: "${k}=${v}") envVars ++ [ "HOME=/root" ];
              Cmd = [ "/bin/entrypoint.sh" ];
              WorkingDir = "/workspace/pilatus-opencv";
            };
          };
          apps.isolated-build = {
            type = "app";
            program = toString (pkgs.writeShellScript containername ''
              ${pkgs.podman}/bin/podman rmi ${containername} || true
              ${pkgs.podman}/bin/podman load \
                --signature-policy ${policy} \
                --input ${inputs.self.packages.${system}.isolated-build}
              ${podmanRun}
            '');
          };
          apps.isolated-nobuild = {
            type = "app";
            program = toString (pkgs.writeShellScript "run-isolated" ''
              set -euo pipefail
              ${podmanRun}
            '');
          };
        };
    };
}
