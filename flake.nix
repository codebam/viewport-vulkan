{
  description = "A Vulkan renderer for Smithay-based Wayland compositors";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      # No package output, deliberately.
      #
      # This is a library crate with a git dependency on Smithay, which
      # buildRustPackage can vendor only with an `outputHashes` entry that has
      # to be rewritten every time the pin moves. Consumers take it as a Cargo
      # dependency and vendor it with their own lock file; what this flake is
      # for is running the tests.
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            rustc
            cargo
            rustfmt
            clippy

            pkg-config
            # drm-sys generates its bindings with bindgen, which needs
            # libclang and the C headers found for it. smithay's backend_drm
            # is what pulls it in.
            rustPlatform.bindgenHook

            libdrm
            libgbm
            libxkbcommon
            wayland
            wayland-protocols
          ];

          # smithay's wayland_frontend links xkbcommon and backend_gbm links
          # libgbm, both at build time.
          LIBRARY_PATH =
            pkgs.lib.makeLibraryPath [ pkgs.libgbm pkgs.libxkbcommon ];

          # ash dlopens libvulkan.so.1. The tests that ask for a device skip
          # themselves when there is no render node, but they have to get as
          # far as the dlopen to do it — without the loader they fail instead.
          shellHook = ''
            export LD_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath [
                pkgs.vulkan-loader
                pkgs.wayland
                pkgs.libxkbcommon
                pkgs.libgbm
              ]
            }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        };
      });
    };
}
