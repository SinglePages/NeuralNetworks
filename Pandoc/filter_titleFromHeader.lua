local title

function starts_with(str, start)
  return str:sub(1, #start) == start
end

function title_from_header (header)

  if header.level > 1 then
    return header
  end

  if not title then
    if starts_with(pandoc.utils.stringify(header), "List of") then
      return {}
    end
    title = header.content
    return {}
  end

  local msg = '[WARNING] title already set; discarding header "%s"\n'
  io.stderr:write(msg:format(pandoc.utils.stringify(header)))
  return {}
end

return {
  {Meta = function (meta) title = meta.title end}, -- init title
  {Header = title_from_header},
  {Meta = function (meta) meta.title = title; return meta end}, -- set title
}
