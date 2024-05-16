--[[
	https://github.com/mochji/synapsea
	core/regularizers.lua

	Synapsea, a simple yet powerful machine learning framework for Lua.
	Copyright (C) 2024 mochji

	MIT License
]]--

local canindex = require("core.utils.canindex")

local regularizersModule = {
	l1,
	l2
}

function regularizersModule.l1(parameters, args)
	local function absoluteSum(tbl)
		local sum = 0

		for a = 1, #tbl do
			if canindex(tbl[a]) then
				sum = sum + absoluteSum(tbl[a])
			else
				sum = sum + math.abs(tbl[a])
			end
		end

		return sum
	end

	local function regularizerFunc(gradient, lambda, l1Norm)
		l1Norm = l1Norm or absoluteSum(gradient)

		for a = 1, #gradient do
			if canindex(args.gradient[a]) then
				gradient[a] = regularizerFunc(gradient[a], lambda, l1Norm)
			else
				gradient[a] = gradient[a] + lambda * l1Norm
			end
		end
	end

	local lambda = args.lambda

	for _, parameter in pairs(parameters) do
		if type(parameter) == "number" then
			parameter = parameter + lambda * math.abs(parameter)
		else
			parameter = regularizerFunc(parameter, lambda)
		end
	end

	return parameters
end

function regularizersModule.l2(parameters, args)
	local function squaredSum(tbl)
		local sum = 0

		for a = 1, #tbl do
			if canindex(tbl[a]) then
				sum = sum + squaredSum(tbl[a])
			else
				sum = sum + tbl[a]^2
			end
		end

		return sum
	end

	local function regularizerFunc(gradient, lambda, l2Norm)
		l2Norm = l2Norm or squaredSum(gradient)

		for a = 1, #gradient do
			if canindex(args.gradient[a]) then
				gradient[a] = regularizerFunc(gradient[a], lambda, l2Norm)
			else
				gradient[a] = gradient[a] + lambda * l2Norm
			end
		end
	end

	local lambda = args.lambda

	for _, parameter in pairs(parameters) do
		if type(parameter) == "number" then
			parameter = parameter + lambda * parameter^2
		else
			parameter = regularizerFunc(parameter)
		end
	end

	return parameters
end

return regularizersModule
