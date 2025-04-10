#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>

#define BUFFER_SIZE 6

typedef int buffer_item;
buffer_item buffer[BUFFER_SIZE];

void *producer(void *arg);
void *consumer(void *arg);

sem_t bin_sem;

int counter = 0;
int index_counter = 0;

char thread1[]="Thread A";
char thread2[]="Thread B";
char thread3[]="Thread C";

int main(int argc, char **argv) {
	pthread_t t1, t2, t3;
	void *thread_result;
	int state;
	
	state = sem_init(&bin_sem, 0, 0);
	
	if(state!=0) {
		puts("Error : semaphore");
	}
	
	// Create thread1, thread2, thread3
	pthread_create(&t1, NULL, producer, &thread1);
	pthread_create(&t2, NULL, consumer, &thread2);
	pthread_create(&t3, NULL, consumer, &thread3);
	
	// Waiting thread to terminate
	pthread_join(t1, &thread_result);
	pthread_join(t2, &thread_result);
	pthread_join(t3, &thread_result);
	
	printf("Terminate => %s, %s, %s!!! \n", thread1, thread2, thread3);
	printf("Final Counter : %d \n", counter);
	
	sem_destroy(&bin_sem);
	return 0;
}

void *producer(void *arg) {
	int i;
	printf("Creating Thread : %s \n", (char*)arg);
	
	for(i=0; i<BUFFER_SIZE; i++) {
		if(index_counter<BUFFER_SIZE) {
			buffer[counter] = counter;
			index_counter++;
			counter++;
			
			printf("%s : INSERT item to BUFFER %d \n", (char*)arg, counter);
			sem_post(&bin_sem);
		}
		else {
			sleep(1);
		}
	}
}

void *consumer(void *arg) {
	int i;
	printf("Creating Thread : %s \n", (char*)arg);
	
	for(i=0; i<BUFFER_SIZE/2; i++) {
		sem_wait(&bin_sem);
		sleep(1);
		
		printf("%s : REMOVE item from BUFFER %d \n", (char*)arg, counter);
		
		index_counter--;
		buffer[counter] = 0;
		counter--;
	}
}
