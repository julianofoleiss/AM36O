# AM36O - Aprendizagem de Máquina

Este é um curso introdutório a nível de graduação que estou ministrando na Universidade Tecnológica Federal do Paraná, Câmpus Campo Mourão.

O objetivo deste material é introduzir os conceitos principais relacionados a tarefas de **Classificação**. O material apresenta os conceitos de forma gradual, focando em boas práticas e técnicas amplamente utilizadas em tarefas de classificação em geral.

Para centralizar a discussão em torno de técnicas de classificação, este material não contempla engenharia de características. Resolvi deixar este assunto de fora pois necessita de conhecimento especialista que não é possível abordar com profundidade em apenas um semestre.

Esse material não deve ser utilizado como referência primária ou exaustiva para nenhum dos assuntos abordados.

Se você usar este material em alguma disciplina ou curso, entre em contato! Gostaria de saber como foi o aproveitamento dos alunos!

## Uso

Você pode usar o Anaconda ou o pip. Este código foi testado com Python 3.8.8, mas deve funcionar com Python 3.5+.

### Conda (testado)

O arquivo ``requirements-conda.txt`` pode ser usado para criar um ambiente conda com os mesmos pacotes usados no desenvolvimento do material. Para isso basta:

``conda create --name <env> --file requirements-conda.txt``

### pip

Se você quiser usar o pip para baixar, recomendo que faça em um ambiente virtual, criado com ``virtualenv`` ou ferramentas similares.

O código deste repositório não está usando nenhuma função deprecada ou experimental. Portanto, deve funcionar com qualquer versão razoavelmente atual das bibliotecas abaixo.

``pip install jupyter numpy scipy matplotlib scikit-learn seaborn``