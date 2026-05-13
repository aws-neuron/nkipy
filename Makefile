.PHONY: install sync relay clean

UV_SYNC_ARGS ?= --all-groups

install: sync relay

sync:
	uv sync $(UV_SYNC_ARGS)

relay:
	$(MAKE) -C relay/src all

clean:
	$(MAKE) -C relay/src clean
