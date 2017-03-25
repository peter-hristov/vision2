function subtile(m,n,i)

h = 1/m;  w = 1/n;

[c r] = ind2sub([n m], i);

subplot('Position',[(c-1)*w (m-r)*h w h]);

axis off
