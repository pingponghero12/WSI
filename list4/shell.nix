{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    cmake
    gcc
    gnumake
    gtest
    doxygen
    (python312.withPackages (
      ps: with ps; [
        numpy
        scipy
        pandas
        matplotlib
        pytest
        scikit-learn
        seaborn
        pybind11
      ]
    ))
  ];
}
