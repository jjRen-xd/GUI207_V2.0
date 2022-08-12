function retn = ToHrrp(sPath,dPath)
    
    retn=0;
    data6 = load(sPath);
    data6 = getfield(data6,'radio101');
    
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
        Xw = X(:,n).*w; % 扫频数据加窗
        x(:,n) = ifftshift(ifft(Xw,N_fft))*point; %IFFT变换到时�?
    end
    x = x./maxRng0; %去除加窗对幅度的影响
    hrrp128 = log(abs(x));
    % x_dB = log(abs(x))/log(20);
    save(dPath,'hrrp128')
    retn=1;
end
