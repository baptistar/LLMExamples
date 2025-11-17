from llm_prompting import LLM
from utils import set_seed
import random

set_seed(0)

# define model
model_name = "Qwen/Qwen2.5-3B-Instruct"
#model_name = "Qwen/Qwen2.5-3B"
QW = LLM(model_name)

# set number of stochastic draws
num_random = 5
QW.temperature = 0.5
QW.max_tokens  = 1000

# add prompting skills
prompting_skills = {"role": "system",  "content":"""Skill <extract_digits>: Extract the digits in a number to a list. 
For example, Extract digits in 123 to D=[1,2,3]. Extract digits in 7654 to D=[7,6,5,4]. \
Skill <list_length>: Get the number of elements in a list.
For example, D=[1,2,3], len(D)=3. A=[1,2,4,5,6], len(A)=5.

Skill <mul_two_single_digit_number>: Multiply two single-digit numbers.
0*1=0 0*2=0 0*3=0 0*4=0 0*5=0 0*6=0 0*7=0 0*8=0 0*9=0
1*1=1 1*2=2 1*3=3 1*4=4 1*5=5 1*6=6 1*7=7 1*8=8 1*9=9
2*1=2 2*2=4 2*3=6 2*4=8 2*5=10 2*6=12 2*7=14 2*8=16 2*9=18
3*1=3 3*2=6 3*3=9 3*4=12 3*5=15 3*6=18 3*7=21 3*8=24 3*9=27
4*1=4 4*2=8 4*3=12 4*4=16 4*5=20 4*6=24 4*7=28 4*8=32 4*9=36
5*1=5 5*2=10 5*3=15 5*4=20 5*5=25 5*6=30 5*7=35 5*8=40 5*9=45
6*1=6 6*2=12 6*3=18 6*4=24 6*5=30 6*6=36 6*7=42 6*8=48 6*9=54
7*1=7 7*2=14 7*3=21 7*4=28 7*5=35 7*6=42 7*7=49 7*8=56 7*9=63
8*1=8 8*2=16 8*3=24 8*4=32 8*5=40 8*6=48 8*7=56 8*8=64 8*9=72
9*1=9 9*2=18 9*3=27 9*4=36 9*5=45 9*6=54 9*7=63 9*8=72 9*9=81
     
Skill <add_multiple_numbers>: Add multiple numbers such as m+n+p:
1. Add the first two numbers m+n and get the result r1=m+n.
2. Add the third number p to r1 and get the result r2=r1+p.
For example, calculate 128+64+79:
1. Add the first two numbers 128+64 and get the result r1=128+64=192.
2. Add the third number 79 to r1 and get the result r2=192+79=271.
So 128+64+79=271.

Example: Calculate 184*67:
Answer:
1. Using Skill <extract_digits> and Skill <list_length>, extract the digits in 184 to DM=[1,8,4].
len(DM)=3. Extract the digits in 67 to DN=[6,7]. len(DN)=2.
2. Add 0,1,len(DM)-1=2 zeros to the end of every number in DM=[1,8,4] according to
the position of the number in DM: DMO=[1*100,8*10,4*1]=[100,80,4].
3. Add 0,len(DN)-1=1 zeros to the end of every number in DN=[6,7] according to the
position of the number in DN: DNO=[6*10,7*1]=[60,7].
4. Using Skill <mul_two_single_digit_number>, multiple every number in DMO=[100,80,4]
with every number in DNO=[60,7] and get R=[100*60,100*7,80*60,80*7,4*60,4*7]=
[6000,700,4800,560,240,28].
5. Using Skill <add_multiple_numbers>, add all the numbers in R=[6000,700,4800,560,240,28],
6000+700+4800+560+240+28:
i. Add the first two numbers: r1=6000+700=6700.
ii. Add the third number 4800 to r1=6700: r2=6700+4800=11500.
iii. Add the fourth number 560 to r2=11500: r3=11500+560=12060.
iv. Add the fifth number 240 to r3=12060: r4=12060+240=12300.
v. Add the sixth number 28 to r4=12300: r5=12300+28=12328.
6. So the answer is 12328"""}

# generate random numbers of length length+1
num1 = '378'
num2 = '64'
task = {"role": "user", "content": f"Calculate {num1}*{num2}"}
print('\n' + task['content'])
print(f'Correct answer: {int(num1)*int(num2)}')
# run sampler
for _ in range(num_random):
    print('======== Stochastic Sampling (No skill) =========')
    QW.generate([task], stochastic=True)
    print('\n')
# run sampler
for _ in range(num_random):
    print('======== Stochastic Sampling (+ skill) =========')
    QW.generate([prompting_skills, task], stochastic=True)
    print('\n')

