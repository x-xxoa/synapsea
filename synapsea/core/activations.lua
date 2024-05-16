--[[
	https://github.com/mochji/synapsea
	core/activations.lua

	Synapsea, a simple yet powerful machine learning framework for Lua.
	Copyright (C) 2024 mochji

	MIT License
]]--

local canindex = require("core.utils.canindex")

local activationsModule = {
	sigmoid,
	tanh,
	relu,
	leakyRelu,
	elu,
	exponential,
	swish,
	binaryStep,
	softmax,
	softplus,
	softsign,
	linear,
	variableLinear,
	hardSigmoid,
	hardTanh
}

function activationsModule.sigmoid(x, derivative)
	if derivative then
		return (1 / (1 + math.exp(-x))) * (1 - (1 / (1 + math.exp(-x))))
	end

	return 1 / (1 + math.exp(-x))
end

function activationsModule.tanh(x, derivative)
	if derivative then
		return 1 - (((math.exp(2 * x) - 1) / (math.exp(2 * x) + 1))^2)
	end

	return math.tanh(x)
end

function activationsModule.relu(x, derivative)
	if derivative then
		if x > 0 then
			return 1
		end

		return 0
	end

	return math.max(0, x)
end

function activationsModule.leakyRelu(x, derivative, alpha)
	if derivative then
		if x > 0 then
			return 1
		end

		return alpha
	end

	if x > 0 then
		return x
	end

	return x * alpha
end

function activationsModule.elu(x, derivative, alpha)
	if derivative then
		if x < 0 then
			return alpha * math.exp(x)
		end

		return 1
	end

	if x < 0 then
		return alpha * (math.exp(x) - 1)
	end

	return x
end

function activationsModule.exponential(x)
	return math.exp(x)
end

function activationsModule.swish(x, derivative, alpha)
	if derivative then
		return (math.exp(-alpha * x) * x + math.exp(-alpha * x) + 1) / ((math.exp(-alpha * x) + 1)^2)
	end

	return x / (1 + math.exp(-alpha * x))
end

function activationsModule.binaryStep(x, derivative)
	if derivative then
		return 0
	end

	if x > 0 then
		return 1
	end

	return 0
end

function activationsModule.softmax(x, derivative)
	local getExpSum, softmax, softmaxDerivative

	getExpSum = function(x)
		local expSum = 0

		for a = 1, #x do
			if canindex(x[a]) then
				expSum = expSum + getExpSum(x[a])
			else
				expSum = expSum + math.exp(x[a])
			end
		end

		return expSum
	end

	softmax = function(x, expSum)
		local output = {}

		for a = 1, #x do
			if canindex(x[a]) then
				output[a] = softmax(x[a], expSum)
			else
				output[a] = math.exp(x[a]) / expSum
			end
		end

		return output
	end

	softmaxDerivative = function(x, expSum)
		local output = {}

		for a = 1, #x do
			if canindex(x[a]) then
				output[a] = softmaxDerivative(x[a], expSum)
			else
				output[a] = (math.exp(x[a]) / expSum) * (1 - (math.exp(x[a]) / expSum))
			end
		end

		return output
	end

	if derivative then
		return softmaxDerivative(x, getExpSum(x))
	end

	return softmax(x, getExpSum(x))
end

function activationsModule.softplus(x, derivative)
	if derivative then
		return 1 / (1 + math.exp(-x))
	end

	return math.log(1 + math.exp(x))
end

function activationsModule.softsign(x, derivative)
	if derivative then
		if x == 0 then
			return 1
		end

		return x / (x * (1 + math.abs(x)^2))
	end

	return x / (1 + math.abs(x))
end

function activationsModule.linear(x, derivative)
	if derivative then
		return 1
	end

	return x
end

function activationsModule.variableLinear(x, derivative, alpha)
	if derivative then
		return alpha
	end

	return x * alpha
end

function activationsModule.hardSigmoid(x, derivative)
	if derivative then
		if x < -2.5 or x > 2.5 then
			return 0
		end

		return 0.2
	end

	return math.max(0, math.min(1, x * 0.2 + 0.5))
end

function activationsModule.hardTanh(x, derivative)
	if derivative then
		if x < -1 or x > 1 then
			return 0
		end

		return 1
	end

	return math.max(0, math.min(1, x * 2))
end

return activationsModule
