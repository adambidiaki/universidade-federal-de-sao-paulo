# Estimação de Altura em Imagens Monoculares

Este repositório reúne os materiais desenvolvidos para o projeto da disciplina de Visão Computacional, incluindo o artigo, a apresentação e os códigos utilizados na avaliação das seguintes abordagens para estimação de altura em imagens monoculares:

* **Geometria Projetiva**, seguindo a metodologia proposta por Criminisi *et al*.

* **Metric3D**, utilizando estimativa monocular de profundidade métrica.

Ambas as abordagens são avaliadas utilizando o mesmo conjunto de dados, permitindo uma comparação direta da precisão e do desempenho de suas estimativas de altura.

---

# Estrutura do Repositório

```texto
height-measurement/
│
├── data/
│ ├── images/
│ └── labels/
│
├── docs/
│ ├── paper.pdf
│ └── presentation.pdf
│
├── Metric3D/
│ ├── model/
│ ├── calibrate.py
│ └── measure.py
│
└── projective-geometry/
  ├── model/
  ├── calibrate.py
  └── measure.py
```

## Descrição das Pastas

### `data/`

Contém o conjunto de dados usado durante os experimentos.

* `images/` imagens usadas para avaliação
* `labels/` arquivos de texto contendo a altura de referência

---

### `docs/`

Contém a documentação relacionada ao projeto.

* Artigo científico
* Apresentação de slides

---

### `Metric3D/`

Contém a implementação baseada no framework Metric3D.

Antes de executar os experimentos, clone o repositório oficial do Metric3D neste diretório:

```bash
cd Metric3D
git clone https://github.com/YvanYin/Metric3D.git
```

Após a clonagem, siga as instruções de instalação do repositório original para configuração do ambiente virtual.

---

### `projective-geometry/`

Contém a implementação baseada em Geometria Projetiva, seguindo a metodologia proposta por Criminisi *et al*.