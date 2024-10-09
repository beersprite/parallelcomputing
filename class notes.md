## OpenMP

- garante threads
- tira poder do user
  
_flag -fopenmp_

`#include <omp.h>`: apenas funcoes da lib, directives n precisa do header

- directives = instruções p o compilador

uma linha que openMP cria e gerencia pthreads por baixo dos panos

```
#pragma omp parallel num_threads(NUMERO)
  {
    //codigo
  }
```

cria numero de threads = n nucleos do pc


90% dos codigos:

`paraellel for` não usa chaves

```
#pragma omp parallel for
 	for (...) ...
```
  
como distribuir? proxima aula de clausulas

default: toma N e divide por num_threads


faz por baixo dos panos

`gomp_parallel() = gnu omp`


### Funções

usaremos mais para os seguintes

numero de threads, thread id, scheduling, timer

#### Env Variables

podemos definir variáveis sem mexer no código

`> export OMP_NUM_THREADS=2`


#### Ferramentas de análise de paralelismo (nao paraleliza, mas aponta trechos)

GPROF

intel Vtune

AMD uProf


nao se paraleliza `printf`/`IO`

`fprintread`, `fscanf`, `fopen`, nao paraleliza com openMP


### clausulas

#### for loop scheduling

Static: distribui as iterações em blocos sobre as threads em uma maneira round-robin

Dynamic: quando uma thread finaliza a computação do seu bloco, recebe a próxima
porção de trabalho.

Guided: mesmo que o dinâmico. Mas, o tamanho do bloco diminui exponencialmente.

Auto: o compilador ou runtime do OpenMP decide ~~o melhor~~ qualquer um para usar.

Runtime: o esquema de escalonamento é definido de acordo com a variável de
ambiente: OMP_SCHEDULE

---

```
#pragma omp for scheduler(/*<guided, static, dynamic>, <chunk_size?>*/)
  for...
```

chunk_size: bloco atribuido a cada vez que thread solicitar

step: Round Robin

---

#### collapse

```
#pragma omp for collapse(/*<number>*/)
  for...
    for...
```

#### master

```
#pragma omp master
  for...
```

similar a `if(pid=0)`

#### single

```
#pragma omp single
  for...
```

  primeira thread a chegar


#### private variables

```
  void addVector(int *A, int *B, int *C){
    int soma = 0;
    #pragma omp parallel for private(soma)
        for(int i = 0; i < N; i++){
            C[i] = A[i] + B[i];
            soma += C[i];
            printf("Thread %d calculating C[%d] = A[%d] + B[%d]\n", omp_get_thread_num(), i, i, i);
        }
    printf("%d\n\n", soma); //prints 0. each thread has its own soma variable. we need a directive to reduce the variables in every thread.
}
```

#### reduction(<operator>: <list_of_variables>)

```
  void addVector(int *A, int *B, int *C){
    int soma = 0;
    #pragma omp parallel for reduction(+:soma) //creates a private variable for each thread and sums them at the end
        for(int i = 0; i < N; i++){
            C[i] = A[i] + B[i];
            soma += C[i];
            printf("Thread %d calculating C[%d] = A[%d] + B[%d]\n", omp_get_thread_num(), i, i, i);
        }
    printf("%d\n\n", soma); //prints soma correctly.
}
```
