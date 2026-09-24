.PHONY: run proceed

run:
	uv run python main.py

proceed:
	@if pgrep -f "index_fandom.py" > /dev/null; then \
		echo "Индексатор уже запущен, смотрю лог"; \
	else \
		nohup uv run python -u index_fandom.py --resume > index_fandom.log 2>&1 & \
	fi; \
	tail -F index_fandom.log
