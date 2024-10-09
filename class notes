## clausulas

### for loop scheduling

Static: distribui as iterações em blocos sobre as threads em uma maneira round-robin

Dynamic: quando uma thread finaliza a computação do seu bloco, recebe a próxima
porção de trabalho.

Guided: mesmo que o dinâmico. Mas, o tamanho do bloco diminui exponencialmente.

Auto: o compilador ou runtime do OpenMP decide qual o melhor para usar.

Runtime: o esquema de escalonamento é definido de acordo com a variável de
ambiente: OMP_SCHEDULE


#pragma omp for scheduler(/*<guided, static, dynamic>, <chunk_size?>*/)

  for


chunk_size: bloco atribuido a cada vez que thread solicitar

step: Round Robin
