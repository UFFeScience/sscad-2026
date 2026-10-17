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

## Oráculos de corretude (K-Means e MatMul)

Executar até o fim não garante que o workflow calcule o resultado certo. A pasta
`oracle/` contém os oráculos aplicados a **todas as 248 especificações de K-Means e
MatMul que executaram** (ER ou ER+), e o veredito de cada uma. Lysozyme não foi
avaliado por oráculo.

**Critérios** (fixados antes de olhar os resultados, validados nos controles abaixo):

- **MatMul** — `‖C − A·B‖_F ≤ ε·‖A‖_F·‖B‖_F`, com ε = 1e-6 (1e-3 em precisão simples),
  usando os operandos que o próprio workflow usou. Quando as matrizes completas nunca
  existem num mesmo processo (execução em blocos), a grade de blocos é reconstruída a
  partir do fluxo de dados das tarefas, e todo bloco de entrada precisa participar do
  produto verificado.
- **K-Means** — os centróides finais são finitos, estão dentro do intervalo dos dados e
  são uma atualização exata de Lloyd (atribuição ao centróide mais próximo + média por
  cluster) dos centróides anteriores sobre **todo** o conjunto de entrada (erro relativo
  ≤ 1e-6), ou uma solução convergida de Lloyd (mais um passo reduz a inércia em no
  máximo 1e-4).

**Como os valores foram obtidos.** Cada especificação foi reexecutada com um profiler
leve (`capture/oracle_capture_hook.py`) que registra, sem alterar o cálculo, as matrizes
mantidas pelas funções do próprio workflow em todos os processos (driver, workers do
Parsl/PyCOMPSs e tarefas do Nextflow). Os 46 casos que a verificação automática não
conseguiu decidir (p.ex. operandos gerados dentro das tarefas e nunca armazenados) foram
decididos executando o código da própria especificação e comparando a saída com `A·B`
ou com a iteração de Lloyd a partir da mesma inicialização (`manual_checks/`).

**Controles.** As 6 implementações de referência (K-Means e MatMul × 3 sistemas) passam;
6 variantes propositalmente erradas das referências Parsl (`run_mutants.py`: B
transposto, pareamento de blocos errado, acumulação perdida, atribuição ao centróide mais
distante, média errada, centróides nunca atualizados) executam sem erro e são todas
rejeitadas.

**Resultado.** 223 passam e **25 falham**: 14 são placeholders que só imprimem uma
mensagem ou reportam números aleatórios, 8 têm defeitos algorítmicos e 3 não executam
como foram gerados. O motivo de cada falha está em `final_verdicts.csv`; o efeito em ER
e ER+ por célula das Tabelas 2 e 3 do artigo está em `table_impact.csv`.

| Arquivo | Conteúdo |
|---|---|
| `final_verdicts.csv` | Veredito por especificação (`PASS`/`FAIL`), se foi automático ou manual, e o motivo |
| `table_impact.csv` | ER e ER+ antes e depois dos oráculos, por tarefa × sistema × modelo × prompt |
| `manifest.csv` | Arquivo exato que executou com sucesso em cada caso (gerado ou corrigido) |
| `oracle.py` | Implementação dos oráculos |
| `run_oracle_exec.py` | Reexecução com o profiler (reusa a lógica de invocação do harness original) |
| `run_reference_exec.py`, `run_mutants.py` | Controles: referências e variantes erradas |
| `build_manifest.py`, `final_verdicts.py` | Montagem do manifesto e dos vereditos finais |
| `capture/` | Profiler carregado em todos os processos Python do workflow |
| `manual_checks/` | Scripts das verificações diretas dos casos decididos manualmente |

Os scripts foram executados a partir do repositório completo do estudo
(`sscad/testes/automation/oracle/`) e usam seus caminhos; aqui estão publicados para
auditoria.
