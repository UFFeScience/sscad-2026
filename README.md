# Can Large Language Models Engineer Parallel Workflows - Scripts

Este diretório reúne todos os scripts de workflow que **funcionaram** (Funcionou? = 1) no
estudo, junto com os arquivos auxiliares que cada script precisa para ser compreendido e
reexecutado. Os scripts são gerados por LLM (Command, Deepseek, GPT, Gemma) mais as
implementações **de referência**, escritas à mão, usadas como ground truth (verdade de
base). A cobertura abrange os três algoritmos (kmeans, matmul, lysozyme) em três
frameworks de workflow (Nextflow, Parsl, PyCOMPSs).

Os arquivos indicados como RQ2 representam nossa Research Question 1, para geração de workflows e os arquivos indicados como RQ7 representam nossa Research Question 2, para tradução de workflows.

## Organização das pastas

    <arm>/<cell>/<script de entrada + arquivos auxiliares>

Pastas de nível superior `<arm>`:

- `reference/`      – as implementações de workflow de referência (escritas por humanos).
- `raw_success/`    – saída da LLM que executou com **zero intervenção humana**.
- `fixed_success/`  – saída da LLM que só executou **após uma correção humana simples**
                      (um ajuste pequeno e mecânico; veja a coluna `notes` do INDEX.csv).

Cada pasta `<cell>` é nomeada `<framework>_<algorithm>` para células de referência, ou
`<model>_<prompt>_<tema>_<framework>_<algorithm>` nos demais casos.

## O que há dentro de cada célula

- O script de workflow de entrada (nome de arquivo original preservado).
- Apenas os arquivos auxiliares que o script de entrada **referencia pelo nome** em seu
  próprio diretório: scripts auxiliares que ele chama (ex.: `run_kmeans.py`, `matmul.py`)
  e arquivos de entrada nomeados que ele lê (ex.: `mat1.txt`, `dataset_noheader.csv`),
  além de `nextflow.config` / `bin/` quando presentes. Arquivos não relacionados que
  apenas coexistem no mesmo diretório de origem (multi-algoritmo) **não** são incluídos
  deliberadamente; arquivos de prompt da LLM e entradas de outros algoritmos mencionados
  apenas em código morto/residual são excluídos.
- Para cada célula de **lysozyme**, o conjunto padrão de entradas do GROMACS é incluído em
  `config/` (ions.mdp, minim.mdp, genion.group, energy.selection) e
  `dataset/` (1AKI.pdb) — são as entradas que os workflows de lysozyme esperam.
- Para cada célula de LLM (`raw_success/` e `fixed_success/`), `prompt.txt` — o prompt
  exato fornecido ao modelo para gerar aquele script. O prompt é definido por
  (RQ, modelo, tipo de prompt, framework) e cobre os três algoritmos, então o mesmo
  `prompt.txt` aparece nas células de kmeans/matmul/lysozyme de um mesmo arm. Células de
  referência não têm prompt (escritas por humanos).

## INDEX.csv

O `INDEX.csv` mapeia cada célula de volta à sua origem: arm, model, prompt, tema,
framework, algorithm, **run**, o caminho original do script de entrada, a pasta de upload,
a lista de arquivos copiados e observações (notes). Use-o para rastrear qualquer pasta até
o repositório de origem.

A coluna `run` registra **qual das 5 execuções (runs) produziu o script que funcionou** e
foi enviado (`N/A` para células de referência). Ela é extraída do caminho original do
script de entrada e conferida contra `analises/analises_workflows_completo.csv`: todo
valor corresponde a uma run cujo `Funcionou?` é 1.
