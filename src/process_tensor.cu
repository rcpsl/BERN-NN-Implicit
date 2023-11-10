extern "C"
__global__ void process_tensor(float* T, float* result, int N, int M, int n)
{
    // Calculate the thread's global index based on block and thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Determine the number of chunks
    int num_chunks = N / n;

    // Step 1: Chunking (implicitly done by indexing)

    // Step 2: Check equality everywhere except the first row
    for(int chunk_1 = 0; chunk_1 < num_chunks; chunk_1++)
    {
        for(int chunk_2 = chunk_1 + 1; chunk_2 < num_chunks; chunk_2++)
        {
            bool equal = true;
            for(int row = 1; row < n && equal; row++)
            {
                for(int col = 0; col < M; col++)
                {
                    if (T[chunk_1 * n * M + row * M + col] != T[chunk_2 * n * M + row * M + col])
                    {
                        equal = false;
                        break;
                    }
                }
            }
            if (equal)
            {
                // Sum the first rows of the two chunks
                for(int col = 0; col < M; col++)
                {
                    T[chunk_1 * n * M + col] += T[chunk_2 * n * M + col];
                }
            }
        }
    }

    // Step 3: Write the results to the result array
    // This is a naive approach where we copy the entire tensor. 
    // An optimized version would only copy distinct chunks.
    result[idx] = T[idx];
}
