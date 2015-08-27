require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset-mnist' --This is a file which simply donwloads the data TODO: modify it for artificial one
require 'pl'
require 'paths'

torch.manualSeed(1234)

torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')
opt = {}
opt.plot = true
opt.batchSize = 10

classes = {'1','2','3','4','5','6','7','8','9','10'}
geometry = {32,32}
model = nn.Sequential()
-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(1, 32, 5, 5)) -- now it is 28x28x32
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(3, 3, 3, 3)) -- now it is 8x8x32

-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(32, 64, 5, 5)) -- now it is 4x4x64
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- now it is 2x2x64
-- stage 3 : standard 2-layer MLP:
model:add(nn.Reshape(64*2*2))
model:add(nn.Linear(64*2*2, 200))
model:add(nn.Tanh())
model:add(nn.Linear(200, #classes))
print(tostring(model))

parameters,gradParameters = model:getParameters()
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

use_full = true
if use_full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 2000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (modify use_fil to use 60000 samples)')
end

trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

_,t = trainData[1][2]:clone():max(1)
print(t)
t:squeeze()
print(t)

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat("logs", 'train.log'))
testLogger = optim.Logger(paths.concat("logs", 'test.log'))


-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. 10 .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end

    
      -- Perform SGD step:
      sgdState = sgdState or {
          learningRate = 0.05,
          momentum = 0,
          learningRateDecay = 5e-7
      }
      optim.sgd(feval, parameters, sgdState)
      
      -- disp progress
      xlua.progress(t, dataset:size())
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat("logs", 'mnist.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end


-- and train!
--
while true do
   -- train/test
   train(trainData)
   test(testData)

   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end

