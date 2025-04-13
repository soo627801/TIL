/*
#include "header1.h"
#include "header2.h"

int func() {
  header1::foo();  // header1 이란 이름 공간에 있는 foo 를 호출
}

// 계속 호출하기 귀찮다면 다음처럼 사용하면 됨.
using header1::foo;
int main() {
  foo();  // header1 에 있는 함수를 호출
}
// 또는
using namespace header1;
int main() {
  foo();  // header1 에 있는 함수를 호출
  bar();  // header1 에 있는 함수를 호출
}   // foo 뿐만 아니라 모든 것을 header1:: 없이 사용 가능
*/

// 마찬가지로 std를 붙이기 귀찮으면
#include <iostream>
using namespace std;

int main() {
  cout << "Hello, World!!" << endl;
  return 0;
}
