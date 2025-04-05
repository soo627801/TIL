#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
	int res;
	res = fork();
	
	if (res < 0) {
		perror("fork");
		exit;
	}
	
	if (res == 0) {
		char *argv[] = {"ls", "-l", NULL};
		printf("I am the child. My pid is %d\n", getpid()); 
		
		res = execv("/bin/ls", argv);
		//res = execl("/bin/ls", "ls", "-l", (char*) 0);
		//res = execvp(argv[0], argv);
		
		if (res == -1) {
			perror("execl");
			exit(2);
		}
		printf("This will never be printed\n");
	}
	
	else {
		int child_pid = res;
		
		printf("I am the father my pid is %d\n", getpid());
		printf("Father is waiting for child to terminate %d\n", child_pid);
		
		waitpid(child_pid, NULL, 0);
		
		printf("Father has seen that the child (%d) exited\n", child_pid);
	}
}
