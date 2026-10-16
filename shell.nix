{ pkgs ? import <nixpkgs> { } }:

let
  llvm = pkgs.llvmPackages;
  clangMajor = pkgs.lib.versions.major (pkgs.lib.getVersion llvm.clang);
in
pkgs.mkShell {
  packages = [
    pkgs.gcc-arm-embedded   # arm-none-eabi-gcc + newlib (string.h/stdlib.h for the ULD)
    llvm.clang
    pkgs.pkg-config
  ];

  shellHook = ''
    unset CC
    unset CXX
  '';

  # bindgen -> libclang.so (clang-tools doesn't ship it)
  LIBCLANG_PATH = "${llvm.libclang.lib}/lib";

  # libclang's own freestanding headers (stdint.h etc.), which Nix's libclang doesn't find by itself
  BINDGEN_EXTRA_CLANG_ARGS = "-isystem ${llvm.libclang.lib}/lib/clang/${clangMajor}/include";
}
