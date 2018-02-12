number = 0

for number in range(10):
   number = number + 1

   if number == 5:
      break    # continue here

   print('Number is ' + str(number))

print('Out of loop')