{ pkgs, lib, config, inputs, ... }:

{
  # Configure Python
  languages.python = {
  	enable = true;
    poetry.enable = true;
  };

  # Add Dev Packages
  packages = [
    pkgs.git
  ];

  # Disable Automatic Cache Management
  cachix.enable = false;
}
