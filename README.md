# gpu-test
Script para rodar e testar a gpu do laboratório MALTA.

# Passos
## 1. Passo
* Fazer o download e a instalação do MiniConda rodando o comando abaixo em um terminal.
  * `sh install_env.sh`
* Fechar o terminal e abrir um novo.
* No novo terminal, rodar o comando:
  * `sh configure_conda.sh`

## 2. Passo
* Fechar o terminal e abrir um novo.
* No novo terminal escrever o seguinte comando: `conda activate test`.
* O terminal deve agora ter a anotação `(test)` antes do nome do usuário.

## 3. Passo
* Começar a treinar a rede neural usando o seguinte comando no mesmo terminal aberto anteriormente:
  * `python teste.py`
* Cuidar de como a placa suporta o treinamento usando o comando `watch nvidia-smi`