
class1 = ( ( (a < 0) - (ytt < 0) ) == 0) % where 1 correct, where 0 incorrect
class2 = ( ( (a < 0) - (ytt < 0) ) ~= 0)

a1 = a(class1)
a2 = a(class2)

for i=1:size(a1)
h(i)=-(a1(i)*log(a1(i)) + (1-a1(i))*log(1-a1(i)))
end
%neeraj = abs(a-ytt)r

%[a ytt a<0 ytt<0 cltzero neeraj]

%neraj(cltzero)
%neeraj(~cltzero)
%classificationRate = 1 - sum(abs((a > 0) - (ytt > 0))) / size(ytt, 1)