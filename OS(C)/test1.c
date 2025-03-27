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

	// Check whether this is the father(parent) or the child
	// the child got 0 from fork.
	// the parent got a positive number from fork
	if (res == 0) {
		char *args[] = {"ls", "-l", NULL};

		// The child enters this block
		printf("I am the child. My pid is %d\n", getpid()); 

		// Make this process run the code of ls.
		// this function will destroy the current process and it will be no more!
		res = execv("/bin/ls", args);

		// the only way we can return from execv is if there was an error
		if (res == -1) {
			perror("execv");
			exit(2);
		}
		// This code is never reached
		printf("This will never be printed\n");
	}
	
	else {
		int child_pid = res;

		// The parent enters this block
		// print the parent pid
		printf("I am the father my pid is %d\n", getpid());
		printf("Father is waiting for child to terminate %d\n", child_pid);

		// Wait for the child to terminate
		waitpid(child_pid, NULL, 0);

		// If the parent exits before the child, the child will become a child of process 0
		// The child is in zombie mode, until its parent performs the wait function call
		// Instead of wait we could set a signal handler for SIGCHLD to be notified of a child termination
		printf("Father has seen that the child (%d) exited\n", child_pid);
	}
}
