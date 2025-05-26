#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 8080

int main(void) {
	int sock;
	struct sockaddr_in addr;
	char buffer[1024];
	const char *msg = "hello world!";
	int recv_len;
	
	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		perror("socket ");
		return 1;
	}
	
	memset(&addr, 0x00, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = inet_addr("127.0.0.1");
	addr.sin_port = htons(PORT);
	
	if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		perror("connect ");
		return 1;
	}
	
	if (send (sock, msg, strlen(msg), 0) < 0) {
		perror("send ");
		return 1;
	}
	
	if ((recv_len = recv(sock, buffer, 1024, 0)) < 0) {
		perror("recv ");
		return 1;
	}
	
	buffer[recv_len] = '\0';
	printf("received data : %s\n", buffer);
	
	close(sock);
	
	return 0;
}
