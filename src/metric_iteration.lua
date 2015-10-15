require 'torch'
require 'math'
require 'optim'

torch.manualSeed(42)

local optimMethod
local optimState
optimMethod = optim.adagrad
optimState = {
    learningRate = 1e-1
}

parameters = {}
parameters.alpha = 1

miniBatchSize = 256
miniBatchID = 1

local feval = function(x)
    local batch_loss = 0
    local gradParameters = 0
    -- loss is the function value and gradParameters is \frac{\nabla f}{\nabla x }
    local active_constraint_set = {}
    for j =miniBatchID, min(miniBatchID+miniBatchSize, dataSetSize) do
        positive_p_dist, positive_pair = posDistMatrix.min(1)
        negative_p_dist, negative_pair = posDistMatrix.min(1)
        active_constraint_set.add()
    end

    for c_id = 1, num_points_source do
        if active[c_id] then
            local w_cur_grad
            local cur_loss = 0
            torch:mm(w_cur_grad, anchor:t(), positive_pair)
            gradParameters:add(w_cur_grad)
            cur_loss:add()
            torch:mm(w_cur_grad, anchor:t(), negative_pair)
            gradParameters:add(-w_cur_grad)
            cur_loss:add()
            cur_loss:add(parameters.alpha)
            batch_loss:add()
        end
    end

    -- add the regularizer loss and gradient
    batch_loss:add(reg_loss)
    gradParameters:add(reg_grad)

    return batch_loss, gradParameters
end


local losses = {}

local iterations = 1000
for i = 1, iterations do
    local _, batch_loss = optimMethod(feval, w_matrix, optimState)

    -- project the matrix to positive-semi-definite cone
    local e,V = torch.eig(w_matrix)
    local e_real = e:select(2, 1)
    e_real[torch.lt(e_real,0)] = 0
    e_real:sqrt() -- take the sqrt to compute vsqrt(e)*vsqrt(e)^T
    V:cmul(l_real:reshape(1,num_dims):expand(num_dims,num_dims))
    toch:mm(w_matrix, V, V:t())

    -- we keep the best matrix since it is subgradient method
    c_value, c_gradients = feval(w_matrix)
    if c_value < best_c then
        best_c = c_value
        best_w_matrix:copy(w_matrix)
    end

end
