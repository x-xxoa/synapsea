--[[
	https://github.com/mochji/synapsea
	core/utils/canindex.lua

	Synapsea, a simple yet powerful machine learning framework for Lua.
	Copyright (C) 2024 mochji

	MIT License
]]--

return function(item)
	return type(item) == "table" or getmetatable(item).__index
end
