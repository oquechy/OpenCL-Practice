__kernel void convolution(__global float *a, __global float *b, __global float *c, int n, int m) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row >= n || col >= n)
        return;

    float sum = 0;
    int hm = (m - 1) / 2;

    for (int k = -hm; k <= hm; ++k) {
        for (int l = -hm; l <= hm; ++l) {
            int i = row + k, j = col + l;
            if (0 <= i && i < n && 0 <= j && j < n)
                sum += a[i * n + j] * b[(hm + k) * m + (hm + l)];
            c[row * n + col] = sum;
        }
    }
}
