fct=@(x) 0.8*sin(x/4)+0.4*sin(pi*x/4)+0.1*cos(pi*x) % funkcja aproksymowana
%fct=@(x) 0.2*(x-4).^2; % funkcja aproksymowana
%dane pomiarowe
xp=[0:0.25:10]';    % wektor wejsc
dp=fct(xp);         % oczekiwany wektor wyjsc
p= length(xp);      % rozmiar wektora wejsc

k=10;               % liczba neuronow ukrytych/centrow; k<=p

% Wyznaczenie centrów funkcji RBF
% centra funkcji RBF w punktach pomiarowych
c=zeros(1,k);   
for i= 1:k  % polozenie centrow od polowy przedzialu p/k (p/(2k)), co p/k
    c(:,i)=xp(floor(i*p/k-(p/(2*k)))); 
end
% redukcja liczby punktow pomiarowych o centra
p = p-k; 
x = setxor(xp,intersect(xp,c));
d = fct(x);

% centra funkcji RBF wybierane losowo z przedzialu danych pomiarowych
% c=(xp(end)-xp(1))*rand(1,k)+xp(1);   % losowe polozenie centrow z d.uczacych
% x=xp;                                % przepisanie punktow pomiarowych
% d=dp;                                
% p=p;

ts=(c(:,1)-c(:,2))'*(c(:,1)-c(:,2));    % kwadrat odleglosci miedzy centrami (metryka euklidesowa)
for i=1:k;                              % znajdz max kwadratu odleglosc miedzy centrami
    for j=i:k
        ts=max(ts,(c(:,i)-c(:,j))'*(c(:,i)-c(:,j))); 
    end
end
sigma = ts/k;      % parametr szerokosci centrow

% radialna funkcja bazowa - gaussa |   -r^2/(2*sigma^2)
phi=@(x,c) exp(-((x-c)'*(x-c))/(2*sigma^2));

% inicjalizacja i wypelnienie macierzy Phi (pobudzenia RBF)
Phi = zeros(p,k+1);     % macierz pobudzen neuronow warstwy ukrytej                       
Phi(:,1)=1;             % wejscie progowe
for i=1:p
    for j=1:k
        Phi(i,j+1)=phi(x(i),c(j));
    end
end
Phi_odwr=(Phi'*Phi)\(Phi');     % macierz pseudoodwrotna macierzy Phi
w=Phi_odwr*d;                   % wyznaczenie wektora wag

%% testowanie sieci na danych ucz¹cych
d_siec = Phi*w;         % wyznacz wyjscie sieci
mse_ucz=immse(d_siec,d);% wyznacz b³¹d w odp. na zestaw ucz¹cy

%% testowanie sieci na zestaw testowy
% podanie zestawu testowego
x_test=[0:0.1:10]';
d_test=fct(x_test);
p_test=length(x_test);

Phi = zeros(p_test,k+1);   % macierz pobudzen RBF
Phi(:,1)=1;                % wejscie progowe w macierzy Phi
for i=1:p_test;            % uzupelnienie macierzy Phi
    for j=1:k
        Phi(i,j+1)=phi(x_test(i),c(j));
    end
end

% odpowiedz sieci na zestaw testowy
d_siec_test = Phi*w;                % wyznacz wyjscie sieci 
mse_test=immse(d_siec_test,d_test)  % MSE dla zestawu testowego

% wykresy
figure(1); hold on;
plot(x_test,d_siec_test,'-b')
plot(x_test,d_test,'-g')
plot(x,d_siec,'-k');
plot(xp,dp,'*r')
plot(c,fct(c),'sk')
str_title=sprintf('RBF. Aproksymacja funkcji')
title(str_title);
legend('odp.sieci na d.testowe','dane testowe(obraz oryginalny)','odp.sieci na d.uczace','dane uczace (wezly aproksymacji)','centra RBF');
