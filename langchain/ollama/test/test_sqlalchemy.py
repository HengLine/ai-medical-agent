try:
    # 在SQLAlchemy 1.4.x版本中，Result类可能在engine.result模块中
    from sqlalchemy.engine.result import Result
    print("SQLAlchemy Result class imported successfully from sqlalchemy.engine.result!")
    print(f"SQLAlchemy version: {__import__('sqlalchemy').__version__}")
    
    # 检查langchain是否使用了不同的导入方式
    try:
        from langchain.vectorstores.chroma import Chroma
        print("Chroma imported successfully from langchain!")
    except ImportError as e:
        print(f"Failed to import Chroma: {e}")

except ImportError as e:
    print(f"Import error: {e}")
    print("Result class might be in a different location in this SQLAlchemy version.")