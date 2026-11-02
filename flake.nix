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
      perSystem = { system, ... }:
        let
          pkgs = import inputs.nixpkgs {
            inherit system;
            config.allowUnfreePredicate = pkg: builtins.elem (inputs.nixpkgs.lib.getName pkg) [ "claude-code" ];
          };
          rust = with inputs.fenix.packages.${system}; combine [
            stable.toolchain
            targets.wasm32-unknown-unknown.stable.rust-std
            targets.aarch64-unknown-linux-gnu.stable.rust-std
          ];
          # `pkgs.clang.cc` is the *unwrapped* clang binary. Unlike the wrapped
          # `pkgs.clang`, it does not know where the C++ standard library and
          # libc headers live, which makes `opencv-binding-generator` fail with
          # `fatal error: 'limits' file not found`. Point clang at the headers
          # from the stdenv toolchain (libstdc++ and glibc) explicitly.
          gcc = pkgs.stdenv.cc.cc;
          gccTarget = pkgs.stdenv.hostPlatform.config;
          gccVersion = pkgs.lib.getVersion gcc;
          gccCxxInclude = "${gcc}/include/c++/${gccVersion}";
          gccIncludeFixed = "${gcc}/lib/gcc/${gccTarget}/${gccVersion}/include-fixed";
          cxxIncludes = pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
            gccCxxInclude
            "${gccCxxInclude}/${gccTarget}"
            gccIncludeFixed
            "${pkgs.stdenv.cc.libc.dev}/include"
          ];
          cIncludes = pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
            gccIncludeFixed
            "${pkgs.stdenv.cc.libc.dev}/include"
          ];
          envVars = {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.clang.cc.lib
              pkgs.opencv
              pkgs.stdenv.cc.cc.lib
            ];
            LIBCLANG_PATH = "${pkgs.clang.cc.lib}/lib";
            CPLUS_INCLUDE_PATH = pkgs.lib.concatStringsSep ":" cxxIncludes;
            C_INCLUDE_PATH = pkgs.lib.concatStringsSep ":" cIncludes;
            PKG_CONFIG_PATH = pkgs.lib.makeSearchPath "lib/pkgconfig" [
              pkgs.glib.dev
              pkgs.aravis.dev
              pkgs.opencv
            ];
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
          };
          commonBuildInputs = [
            rust
            # Use the unwrapped clang: the wrapper provides its own bin/cc and
            # bin/c++, which collide with stdenv.cc inside buildEnv.
            pkgs.clang.cc
            pkgs.stdenv.cc
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
          # Mirrors rust-analyzer-lsp@claude-plugins-official, which only declares the server.
          # Built into the image, because /root/.claude isn't persisted between runs.
          rustAnalyzerLspPlugin = pkgs.writeTextDir "rust-analyzer-lsp/.claude-plugin/plugin.json"
            (builtins.toJSON {
              name = "rust-analyzer-lsp";
              description = "Rust language server for code intelligence and analysis";
              version = "1.0.0";
              lspServers.rust-analyzer = {
                command = "${rust}/bin/rust-analyzer";
                extensionToLanguage.".rs" = "rust";
              };
            });
          claudeCodeWithLsp = pkgs.writeShellScriptBin "claude" ''
            exec ${pkgs.claude-code}/bin/claude --plugin-dir ${rustAnalyzerLspPlugin}/rust-analyzer-lsp "$@"
          '';
          policy = pkgs.writeText "policy.json" ''{"default":[{"type":"insecureAcceptAnything"}]}'';
          containername = "pilatus-opencv-isolated-dev";
          podmanRun = ''
            workspace="$(realpath "''${1:-.}")"
            exec ${pkgs.podman}/bin/podman run --rm -it \
              --network=pasta \
              --tmpfs /tmp \
              -v "$workspace:/workspace/pilatus-opencv:z" \
              -v "../pilatus:/workspace/pilatus:z" \
              -v "../imbuf:/workspace/imbuf:z" \
              -w "/workspace/pilatus-opencv" \
              -e HOME=/root \
              ${containername}:latest /bin/entrypoint.sh
          '';
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
                claudeCodeWithLsp
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
