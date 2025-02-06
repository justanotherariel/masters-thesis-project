{ pkgs, lib, config, inputs, ... }:

{
  # Configure Python
  languages.python = {
  	enable = true;
    venv.enable = true;
    # poetry.enable = true;
    # poetry.activate.enable = false;
    # poetry.install.enable = false;
  };

  # Add Dev Packages
  packages = [
    pkgs.git
    pkgs.ruff
    pkgs.zstd
    pkgs.libdrm
    pkgs.zlib
  ];

  # Disable Automatic Cache Management
  cachix.enable = false;

  # Dev Script
  scripts.project = {
    exec = ''
      # Parse Arguments
      if [ "$#" -eq 0 ]; then
        echo "Usage: project [lint|test]"
        exit 1
      fi

      # Run Linting
      if [ "$1" = "ruff" ]; then
        ruff check src/ tests/ --fix
        ruff format src/ tests/
      fi

      # Run Tests
      if [ "$1" = "test" ]; then
        pytest
      fi
    '';
    description = "Run Linting and Tests";
  };
}
