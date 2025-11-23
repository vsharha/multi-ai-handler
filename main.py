from multi_ai_handler import request_ai, MultiAIHandler



def main():
    client = MultiAIHandler()

    for provider, models in client.list_models().items():
        print(provider)
        print(models)

if __name__ == "__main__":
    main()
