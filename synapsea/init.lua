--[[
	Synapsea v2.0.0-development

	Read the README.md file for documentation and information, 

	https://github.com/mochji/synapsea
	init.lua

	Synapsea, simple yet powerful machine learning framework for Lua.
	Copyright (C) 2024 mochji

	MIT License
]]--

local synapseaPath =
	debug.getinfo(1, "S").source
		:sub(2)
		:match("(.*" .. package.config:sub(1, 1) .. ")")
		or "." .. package.config:sub(1, 1)

local oldPackagePath = package.path
package.path = synapseaPath .. "?.lua"

local synapsea = {
	path         = synapseaPath,
	version      = "v2.0.0-development",

	activations  = require("core.activations"),
	losses       = require("core.losses"),
	initializers = require("core.initializers"),
	optimizers   = require("core.optimizers"),
	regularizers = require("core.regularizers"),
	layers       = require("core.layers.layers"),
	Sequential   = require("core.model.Sequential")
}

for layerName, layerFunc in pairs(synapsea.layers) do
	local buildModule    = require("core.layers.build")
	local errorModule    = require("core.layers.error")
	local gradientModule = require("core.layers.gradient")

	synapsea.layers[layerName] = setmetatable(
		{
			build    = buildModule[layerName],
			error    = errorModule[layerName],
			gradient = gradientModule[layerName]
		},
		{
			__call = function(_, args)
				return layerFunc(args)
			end
		}
	)
end

package.path = oldPackagePath

if synapsea.version:match("development") then
	io.write("\27[1m\27[33mWARNING:\27[0m You are using a development release of Synapsea!\n")
	io.flush()
end

return synapsea
