#!/bin/bash

# pip install
python3 -m pip install -r requirements.txt

# Visual Studio Code :: Package list
pkglist=(
ms-python.python
tabnine.tabnine-vscode
njpwerner.autodocstring
oderwat.indent-rainbow
)
for i in ${pkglist[@]}; do
  code --install-extension $i
done