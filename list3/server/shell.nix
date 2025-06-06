{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = [
    pkgs.gcc
    pkgs.gsl
    pkgs.cmake
  ];
}
