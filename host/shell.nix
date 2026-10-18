{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: [
      ps.pyserial
      ps.numpy
      ps.matplotlib
      ps.tkinter   # Matplotlib GUI backend fallback
    ]))
  ];

  shellHook = ''
    echo "Python environment loaded (pyserial, numpy, matplotlib)."
    echo "Run: python live_grid.py"
  '';
}