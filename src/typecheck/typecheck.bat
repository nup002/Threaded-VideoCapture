call conda activate threaded-VideoCapture
SET file=%1
IF "%~1" == "" (SET file="../threaded_videocapture/main.py")
mypy --disallow-untyped-defs --disallow-incomplete-defs^
 --warn-redundant-casts --warn-unused-ignores --warn-return-any^
 --warn-unreachable --pretty --config-file=mypy.ini^
 --show-error-codes --install-types ./%file%
pause
