## OpenMP

- Garante threads
- Tira poder do user
- Não altera complexidade
  
_flag -fopenmp_

`#include <omp.h>`: apenas funções da lib, _directives_ não precisam do header.

Similar a _<math.h>_, necesssária apenas para funções específicas

- directives = instruções p o compilador

uma linha que openMP cria e gerencia pthreads por baixo dos panos

```
#pragma omp parallel num_threads(NUMERO)
  {
    // código
  }
```

Cria numero de threads = n núcleos do pc


90% dos codigos:

`paraellel for` não usa chaves

```
#pragma omp parallel for
 	for (...) ...
```
  
Como distribuir? Vide [Cláusulas](#cláusulas)

Default: toma N e divide por num_threads


Por baixo dos panos...

`gomp_parallel() = gnu omp`


### Funções

Usaremos mais os seguintes:

`número de threads, thread id, scheduling, timer`

#### Environment Variables

Podemos definir variáveis sem mexer no código

`> export OMP_NUM_THREADS=2`


#### Ferramentas de análise de paralelismo

Não paralelizam, mas apontam trechos que podem ser paralelizados

`GPROF`

`Intel Vtune`

`AMD uProf`

Em geral, não se paraleliza `printf`/`IO`

`fprintread`, `fscanf`, `fopen`, não paraleliza com openMP


### Cláusulas

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

Pode levar à condição de corrida, pode combinar com diretivas de sincronização.

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

#### omp sections
- pode transformar for em sections (manual), como se `omp section` fossem `pthreads`
- não usamos em OpenMP


### Diretivas de Sincronização

#### pragma omp critical

Creates a critical region (mutex)

```
#pragma omp critical
{ // lock
SC;
} // unlock
```

#### pragma omp atomic

Creates an atomic operation in *hardware*. not the same as critical region, which uses mutex.

**because it's in hardware, it is always faster than mutex.**

```
#pragma omp atomic
{
soma +- partial;
}
```

---

Sobre o trabalho:

- Desconsiderar warmup
- 5-6 problemas por maratona
- source code tem entradas e pode usar próprias entradas. (1 trivial, 2 ok, 2 dificeis com resultado irrisório ~5% "perda de tempo" com ponteiros p tudo)
- escolha de tema e grupos no moodle, FIFS
- apresentação, nao tem relatorio. (funções que foram parelizadas com o quê, e os resultados obtidos)
- usar pc santos dummont (40 cores)

  OU

- maratona SSCAD - 3 pessoas + coach no dia 23/24
- recebe por email as infos

---

## CUDA

```nvcc```

---

## Open ACC

```nvc```

Host memory (1tb, eg) > Device Memory (12 gb, eg)

Poder de processamento do Device é maior.

- Incremental assim como OpenMP
- pragma com anotações para expor o paralelismo

- Baixa curva de aprendizado

  ```#pragma acc kernels```

  compilador faz o paralelismo, se possível. se não for possível, imprime log com motivo. o desempenho não é necessariamente melhor.

### How GPUs work

Composto de cuda cores (ou outro nome comercial) - ALU sem branch prediction

Cores de INT, FLOAT, 32, 64 bit. Indicado por cores

Unidades Tensor para multiplicação de matrizes

WARP: conjunto de 32 threads por Streaming M., que executam por ciclo em cada Stream. M. AMD usa 64 threads, devido à arquitetura.

Ideal = threads > cores

| | | |
| --|-- |-- |
| Streaming Multiprocessors | Core + Cache L0 | Shared Memory |
| | | |
| L2 | DDR Global Memory|

Single instruction por todos os cores, alterando apenas a variável usada.


`Profile the code` - quando não saber o que fazer. similar a gprof, mas para nvidia

