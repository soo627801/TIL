#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
// #include <stdlib.h>

int main(int argc, char **argv) {
	int res;		// duplicate the current process
	res = fork();		// fok was successful there are now two processes at this point
	
	// check whether fork is successful
	if (res < 0) {
		perror("fork");
		exit;
	}

	// Check whether this is the father or the child
	// the child got 0 from fork.
	// the parent got a positive number from fork
	if (res == 0) {
		// The child enters this block
		printf("I am the child. My pid is %d\n", getpid()); 
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
		// A child that finishes its execution is in zombie mode,
		// until its parent performs the wait function call
		// Instead of wait we could set a signal handler for SIGCHLD
		// to be notified of a child termination
		printf("Father has seen that the child (%d) exited\n", child_pid);
	}
}
