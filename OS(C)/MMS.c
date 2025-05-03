#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <semaphore.h>
#include <stdint.h>
#include <time.h>

#define MAX_NO_THREADS 20
#define MAX_MEM_SIZE 86
#define MAX_THREAD_MEM 15

typedef struct {
    intptr_t thread_id;
    int size;
    char *addr;
    int active;
} MyThread;

typedef struct {
    char *B_Begin;
    int B_size;
    int MAssign;
} MemoryBlock;

void *a;
char *Start_F;
char *End_F;

MyThread Threads[MAX_NO_THREADS];
MemoryBlock Blocks[MAX_THREAD_MEM];

pthread_mutex_t Mutex;
sem_t bin_sem;
pthread_mutex_t mutx;

int ch;
int No_of_Threads;
int T_Blocks;
int T_indx = 0;

void InitializeBlocks();
void Print_Mem();
void Memory_Info();
void *MU(void *arg);
char *FirstFit(int size);
char *BestFit(int size);
char *WorstFit(int size);
void memory_free(char *initial, int size);
void random_memory_free();

int main() {
    printf("Enter number of threads (1-%d): ", MAX_NO_THREADS);
    scanf("%d", &No_of_Threads);
    if (No_of_Threads < 1 || No_of_Threads > MAX_NO_THREADS) {
        printf("Invalid number of threads.\n");
        return -1;
    }

    printf("Select allocation strategy:\n");
    printf("1. First Fit\n2. Best Fit\n3. Worst Fit\nEnter choice (1~3): ");
    scanf("%d", &ch);
    if (ch < 1 || ch > 3) {
        printf("Invalid strategy.\n");
        return -1;
    }

    srand(time(NULL));

    a = malloc(MAX_MEM_SIZE);
    if (!a) {
        perror("malloc failed");
        exit(1);
    }
    Start_F = (char *)a;
    End_F = Start_F + MAX_MEM_SIZE;

    InitializeBlocks();
    pthread_mutex_init(&mutx, NULL);
    pthread_mutex_init(&Mutex, NULL);
    sem_init(&bin_sem, 0 ,0);

    for (int i = 0; i < No_of_Threads; i++) {
        pthread_t tid;
        pthread_create(&tid, NULL, MU, NULL);
        usleep(100000);
    }

    sleep(15);
    free(a);
    return 0;
}

void *MU(void *arg) {
    char *assigned_val;
    int Bk_Assigned = 2;
    int t_idx;

    pthread_mutex_lock(&mutx);
    t_idx = T_indx;
    Threads[t_idx].thread_id = (intptr_t)pthread_self();
    Threads[t_idx].size = (rand() % 15) + 1;
    Threads[t_idx].active = 0;
    T_indx++;
    pthread_mutex_unlock(&mutx);

    while (Threads[t_idx].size > Bk_Assigned)
        Bk_Assigned *= 2;

    pthread_mutex_lock(&Mutex);
    Memory_Info();

    printf("\n[Thread %d] Requested: %d bytes\n", t_idx + 1, Bk_Assigned);
    printf("Current Memory State Before Allocation:\n");
    Print_Mem();

    // 초반에는 계속 할당 유도
    if (t_idx >= MAX_NO_THREADS / 2 && t_idx % 2 == 0) {
        random_memory_free();
        Memory_Info(); // defragmentation 비활성화로 변경
    }

    switch (ch) {
        case 1: assigned_val = FirstFit(Bk_Assigned); break;
        case 2: assigned_val = BestFit(Bk_Assigned); break;
        case 3: assigned_val = WorstFit(Bk_Assigned); break;
        default: assigned_val = NULL;
    }

    if (assigned_val == NULL) {
        printf("[Thread %d] NO: Allocation failed (request: %d bytes)\n", t_idx + 1, Bk_Assigned);
        pthread_mutex_unlock(&Mutex);
        return NULL;
    }

    Threads[t_idx].addr = assigned_val;
    Threads[t_idx].active = 1;
    for (int i = 0; i < Bk_Assigned; i++)
        assigned_val[i] = (char)(t_idx + 1);

    Memory_Info();
    printf("\n[Thread %d] OK: Allocated %d bytes using %s Fit\n", t_idx + 1, Bk_Assigned,
           ch == 1 ? "First" : ch == 2 ? "Best" : "Worst");
    Print_Mem();
    pthread_mutex_unlock(&Mutex);

    sleep(rand() % 5 + 1);

    pthread_mutex_lock(&Mutex);
    if (Threads[t_idx].active) {
        memory_free(Threads[t_idx].addr, Bk_Assigned);
        Threads[t_idx].active = 0;
    }
    pthread_mutex_unlock(&Mutex);

    return NULL;
}

void random_memory_free() {
    for (int i = 0; i < T_indx; i++) {
        if (Threads[i].active && (rand() % 2 == 0)) {
            printf("\n [Auto Free] Releasing memory from Thread %d\n", i + 1);
            memory_free(Threads[i].addr, Threads[i].size);
            Threads[i].active = 0;
        }
    }
}

char *FirstFit(int size) {
    for (int i = 0; i < T_Blocks; i++) {
        if (!Blocks[i].MAssign && Blocks[i].B_size >= size)
            return Blocks[i].B_Begin;
    }
    return NULL;
}

char *BestFit(int size) {
    int min_diff = MAX_MEM_SIZE + 1;
    char *best = NULL;
    for (int i = 0; i < T_Blocks; i++) {
        if (!Blocks[i].MAssign && Blocks[i].B_size >= size) {
            int diff = Blocks[i].B_size - size;
            if (diff < min_diff) {
                min_diff = diff;
                best = Blocks[i].B_Begin;
            }
        }
    }
    return best;
}

char *WorstFit(int size) {
    int max_size = -1;
    char *worst = NULL;
    for (int i = 0; i < T_Blocks; i++) {
        if (!Blocks[i].MAssign && Blocks[i].B_size >= size) {
            if (Blocks[i].B_size > max_size) {
                max_size = Blocks[i].B_size;
                worst = Blocks[i].B_Begin;
            }
        }
    }
    return worst;
}

void memory_free(char *initial, int size) {
    for (int i = 0; i < size; i++) {
        initial[i] = 0;
    }
    Memory_Info();
    printf("\n[Thread] Memory Freed (%d bytes)\n", size);
    Print_Mem();
}

void InitializeBlocks() {
    for (int i = 0; i < MAX_THREAD_MEM; i++) {
        Blocks[i].B_size = 0;
        Blocks[i].MAssign = 0;
        Blocks[i].B_Begin = Start_F;
    }
    for (int i = 0; i < MAX_MEM_SIZE; i++)
        Start_F[i] = 0;
}

void Memory_Info() {
    T_Blocks = 0;
    char *indx = Start_F;
    char *B_Begin;

    while (indx < End_F) {
        B_Begin = indx;
        if (*indx == 0) {
            while (*indx == 0 && indx < End_F) indx++;
        } else {
            char val = *indx;
            while (*indx == val && indx < End_F) indx++;
        }
        Blocks[T_Blocks].B_Begin = B_Begin;
        Blocks[T_Blocks].B_size = indx - B_Begin;
        Blocks[T_Blocks].MAssign = (*B_Begin != 0);
        T_Blocks++;
    }
}

void Print_Mem() {
    printf("\n Memory State:\n");
    for (int i = 0; i < MAX_MEM_SIZE; i++) {
        printf("%d ", Start_F[i]);
    }
    printf("\n Blocks: %d\n", T_Blocks);
}
