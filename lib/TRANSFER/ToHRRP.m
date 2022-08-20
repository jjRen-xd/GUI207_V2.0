function retn = ToHrrp(sPath,dPath,sName)

    retn=0;
    data6 = load(sPath);
    data6 = getfield(data6,sName);
    
    X = data6;
    
    mid =size(X);
    down_list = 1:1:mid(1);
    X = X(down_list,:);
    
    [frqNum,dataNum] = size(X) 
    win = 'hamming';
    N_fft = 2^nextpow2(frqNum);
    
    point = N_fft/frqNum;
    
    w = window(win, frqNum);
    Rng0 = ifftshift(ifft(w,N_fft))*point;
    maxRng0 = max(abs(Rng0));
    
    x = zeros(N_fft, dataNum);
    for n = 1:dataNum
        Xw = X(:,n).*w;
        x(:,n) = ifftshift(ifft(Xw,N_fft))*point;
    end
    x = x./maxRng0;
    hrrp = log(abs(x));
    save(dPath,'hrrp')
    retn=1;
end
