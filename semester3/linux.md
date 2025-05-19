## Commands  
    find -name name.txt
    find -name *.txt  

## Number of lines
    wc -l access.log

## search lines with grep
    grep "81.143.211.90" access.log
    grep "phpmyadmin2017" access.log

| Symbol / Operator | Description                                                                                                                               |
| :---------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| `&`               | This operator allows you to run commands in the background of your terminal.                                                              |
| `&&`              | This operator allows you to combine multiple commands together in one line of your terminal. it's worth noting that command2 will only run if command1 was successful |
| `>`               | This operator is a redirector - meaning that we can take the output from a command (such as using `cat` to output a file) and direct it elsewhere. |
| `>>`              | This operator does the same function of the `>` operator but appends the output rather than replacing (meaning nothing is overwritten).        |
