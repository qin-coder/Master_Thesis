call mvn clean -Dmaven.test.skip=true
call mvn install -Dmaven.test.skip=true
call mvn compile -Dmaven.test.skip=true
call mvn package -Dmaven.test.skip=true
@echo copy to C:\Users\Administrator\Documents\exec\exec?
pause
copy shaded\target\evosuite-shaded-1.2.1-SNAPSHOT.jar C:\Users\Administrator\Documents\exec\exec\evosuite-shaded-1.2.1-SNAPSHOT.jar
@echo copied
pause