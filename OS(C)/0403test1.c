#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

int index_counter = 0;
void *thread_Insert(void *arg);

char thread1[] = "Thread A";
char thread2[] = "Thread B";

int main(int argc, char **argv) {
	pthread_t t1, t2;
	void *thread_result;

	pthread_create(&t1, NULL, thread_Insert, &thread1);
	pthread_create(&t2, NULL, thread_Insert, &thread2);

	pthread_join(t1, &thread_result);

	pthread_join(t2, &thread_result);

	printf("Terminate => %s, %s!!!\n", &thread1, &thread2);
	printf("Final Index : %d\n", index_counter);

	return 0;
}

void *thread_Insert(void *arg) {
	int i;
	printf("Creating Thread : %s\n", (char*)arg);
	for(i=0; i<20; i++) {
		index_counter++;
	
		printf("%s : INSERT item to number = %d\n", (char*)arg, index_counter);
	}

}
