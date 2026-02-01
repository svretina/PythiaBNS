.PHONY: docs clean

docs:
	uv run pdoc src/pythiabns -o docs/html

clean:
	rm -rf docs/html
